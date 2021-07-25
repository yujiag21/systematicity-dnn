    public void bad() throws Throwable
    {
        String data;

        /* POTENTIAL FLAW: data is null */
        data = null;

        /* POTENTIAL FLAW: null dereference will occur if data is null */
        IO.writeLine("" + data.length());

    }
