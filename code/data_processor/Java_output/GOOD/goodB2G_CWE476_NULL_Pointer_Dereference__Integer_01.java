    private void goodB2G() throws Throwable
    {
        Integer data;

        /* POTENTIAL FLAW: data is null */
        data = null;

        /* FIX: validate that data is non-null */
        if (data != null)
        {
            IO.writeLine("" + data.toString());
        }
        else
        {
            IO.writeLine("data is null");
        }

    }
